import {
  ConflictException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';

import { PrismaService } from 'src/prisma.service';
import { LogService } from 'src/log/log.service';

import {
  CreateManufacturerDto,
  ManufacturerDto,
  UpdateManufacturerDto,
} from './dto/manufacturers.dto';

@Injectable()
export class ManufacturersService {
  constructor(
    private prisma: PrismaService,
    private logService: LogService,
  ) {}

  async findAll(): Promise<ManufacturerDto[]> {
    return this.prisma.manufacturer.findMany();
  }

  async findById(id: string): Promise<ManufacturerDto> {
    const manufacturer = await this.prisma.manufacturer.findUnique({
      where: { id },
    });

    if (!manufacturer) {
      throw new NotFoundException('Manufacturer not found');
    }

    return manufacturer;
  }

  async create(
    userId: string,
    createManufacturerDto: CreateManufacturerDto,
  ): Promise<ManufacturerDto> {
    const createdManufacturer = await this.prisma.manufacturer.create({
      data: createManufacturerDto,
    });

    await this.logService.create({
      userId,
      targetId: createdManufacturer.id,
      targetName: createdManufacturer.name,
      type: 'manufacturer',
      action: 'create',
    });

    return createdManufacturer;
  }

  async update(
    userId: string,
    id: string,
    updateManufacturerDto: UpdateManufacturerDto,
  ): Promise<ManufacturerDto> {
    const existingManufacturer = await this.prisma.manufacturer.findUnique({
      where: { id },
    });

    if (!existingManufacturer) {
      throw new NotFoundException('Manufacturer not found');
    }

    const updatedManufacturer = await this.prisma.manufacturer.update({
      where: { id },
      data: updateManufacturerDto,
    });

    await this.logService.create({
      userId,
      targetId: updatedManufacturer.id,
      targetName: updatedManufacturer.name,
      type: 'manufacturer',
      action: 'update',
    });

    return updatedManufacturer;
  }

  async delete(userId: string, id: string): Promise<ManufacturerDto> {
    const existingManufacturers = await this.prisma.manufacturer.findUnique({
      where: { id },
    });

    if (!existingManufacturers) {
      throw new NotFoundException('Manufacturer not found');
    }

    try {
      const deletedManufacturers = await this.prisma.manufacturer.delete({
        where: { id },
      });

      await this.logService.create({
        userId,
        targetId: deletedManufacturers.id,
        targetName: deletedManufacturers.name,
        type: 'manufacturer',
        action: 'delete',
      });

      return deletedManufacturers;
    } catch {
      throw new ConflictException(
        'Manufacturer is being used in a device and cannot be deleted.',
      );
    }
  }
}
