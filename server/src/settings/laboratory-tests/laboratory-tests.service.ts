import {
  ConflictException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';

import { PrismaService } from 'src/prisma.service';
import { BiomarkersService } from '../biomarkers/biomarkers.service';
import { LogService } from 'src/log/log.service';

import {
  CreateLaboratoryTestDto,
  LaboratoryTestDto,
  UpdateLaboratoryTestDto,
} from './dto/laboratory-test.dto';

@Injectable()
export class LaboratoryTestsService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly biomarkersService: BiomarkersService,
    private logService: LogService,
  ) {}

  async findAll(): Promise<LaboratoryTestDto[]> {
    return this.prisma.laboratoryTest.findMany({
      include: {
        biomarkers: true,
      },
    });
  }

  async findById(id: string): Promise<LaboratoryTestDto> {
    const laboratoryTest = await this.prisma.laboratoryTest.findUnique({
      where: { id },
      include: {
        biomarkers: true,
      },
    });

    if (!laboratoryTest) {
      throw new NotFoundException('Laboratory Test not found');
    }

    return laboratoryTest;
  }

  async create(
    userId: string,
    createLaboratoryTestDto: CreateLaboratoryTestDto,
  ): Promise<LaboratoryTestDto> {
    const { biomarkers, ...dto } = createLaboratoryTestDto;

    await Promise.all(
      biomarkers.map(async (biomarkerId) => {
        await this.biomarkersService.findById(biomarkerId);
      }),
    );

    const createdLaboratoryTest = await this.prisma.laboratoryTest.create({
      data: {
        ...dto,
        biomarkers: {
          connect: biomarkers.map((id) => ({ id })),
        },
      },
      include: { biomarkers: true },
    });

    await this.logService.create({
      userId,
      targetId: createdLaboratoryTest.id,
      targetName: createdLaboratoryTest.name,
      type: 'laboratory-test',
      action: 'create',
    });

    return createdLaboratoryTest;
  }

  async update(
    userId: string,
    id: string,
    updateLabTest: UpdateLaboratoryTestDto,
  ): Promise<LaboratoryTestDto> {
    const { biomarkers, ...dto } = updateLabTest;

    const laboratoryTest = await this.findById(id);

    await Promise.all(
      biomarkers.map(async (biomarkerId) => {
        await this.biomarkersService.findById(biomarkerId);
      }),
    );

    const updateLaboratoryTest = await this.prisma.laboratoryTest.update({
      where: { id },
      data: {
        ...dto,
        biomarkers: {
          disconnect: laboratoryTest.biomarkers.map((b) => ({ id: b.id })),
          connect: biomarkers.map((id) => ({ id })),
        },
      },
      include: {
        biomarkers: true,
      },
    });

    await this.logService.create({
      userId,
      targetId: updateLaboratoryTest.id,
      targetName: updateLaboratoryTest.name,
      type: 'laboratory-test',
      action: 'update',
    });

    return updateLaboratoryTest;
  }

  async delete(userId: string, id: string): Promise<LaboratoryTestDto> {
    await this.findById(id);

    const isLaboratoryTestUsed =
      await this.prisma.appointmentServices.findFirst({
        where: {
          labWorks: {
            some: {
              id,
            },
          },
        },
      });

    if (isLaboratoryTestUsed) {
      throw new ConflictException(
        'Laboratory test is being used and cannot be deleted.',
      );
    }

    const deletedLaboratoryTest = await this.prisma.laboratoryTest.delete({
      where: { id },
      include: { biomarkers: true },
    });

    await this.logService.create({
      userId,
      targetId: deletedLaboratoryTest.id,
      targetName: deletedLaboratoryTest.name,
      type: 'laboratory-test',
      action: 'delete',
    });

    return deletedLaboratoryTest;
  }
}
